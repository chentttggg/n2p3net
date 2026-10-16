from __future__ import annotations

import copy
import subprocess
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

from data.artifact import FoldLocalArtifactPolicy
from data.brainsync import load_brainsync_inference_session, load_brainsync_sessions
from data.domain import adapt_observed_montage
from experiments.brainsync_model_task import inference_record, inference_trial_counts
from experiments.brainsync_model_task import main as model_task
from experiments.fit_brainsync_eog import fit_brainsync_eog_sessions
from tests import PROJECT_ROOT
from tests.support.brainsync import write_bids_session
from tests.support.checkpoints import write_checkpoint
from tests.support.datasets import binary_dataset
from tests.support.io import read_json, write_json
from tests.support.quality import fit_raw_quality
from transfer.binary import target_shrinkage_scores
from transfer.calibrated import (
    CalibrationRecipe,
    fit_calibrated_predictor,
    load_calibrated_predictor,
    load_calibration_source_checkpoint,
    preflight_calibrated_predictor,
    save_calibrated_predictor,
)
from transfer.quality import fit_epoch_quality
from transfer.recipes import TargetRecipe


def _cz_view(dataset):
    return adapt_observed_montage(
        dataset,
        tuple(name for name in dataset.channel_names if name != "CZ"),
        reference_channel="Cz",
    )


def _write_cz_checkpoint(path, dataset):
    return write_checkpoint(path, _cz_view(dataset), source_snapshot_sha256="f" * 64)


def test_inference_admission_counts_cover_preprocessing_and_model_exclusions():
    result = inference_trial_counts(scheduled_trials=450, input_epochs=273, usable_predictions=229)
    assert result["preprocessing_unavailable_trials"] == 177
    assert result["model_unavailable_trials"] == 44
    assert result["unavailable_trials"] == 221
    assert result["evidence_trials"] + result["unavailable_trials"] == result["scheduled_trials"]
    with pytest.raises(ValueError, match="reconcile"):
        inference_trial_counts(scheduled_trials=45, input_epochs=46, usable_predictions=10)


def test_cli_unavailable_cuda_fails_before_fitting_or_loading_model(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "unavailable"
    with pytest.raises(SystemExit):
        model_task(["--task", "inspect", "--device", "cuda", "--model-artifact", str(tmp_path / "absent.pt"),
                    "--output-dir", str(output)])
    result = read_json(output / "summary.json")
    assert result["status"] == "failed" and "CUDA was requested" in result["error"]["message"]


def test_model_task_auto_device_uses_shared_selection(monkeypatch):
    import experiments.brainsync_model_task as task

    monkeypatch.setattr(task, "get_device", lambda: torch.device("cpu"))
    assert task.model_task_device("auto") == "cpu"


def test_changed_frequency_is_frozen_through_cli_calibration_and_raw_inference(tmp_path, sessions):
    from dataclasses import asdict

    import experiments.brainsync_model_task as task
    from data.brainsync import BRAIN_SYNC_PREPROCESSING

    roots, testing = sessions
    spec = replace(BRAIN_SYNC_PREPROCESSING, l_freq=.5)
    config = tmp_path / "preprocessing.json"
    write_json(config, asdict(spec))
    artifact = tmp_path / "frequency.pt"
    task.main(["--task", "calibrate", "--session-dir", str(roots[0]),
               "--model-artifact", str(artifact), "--output-dir", str(tmp_path / "fit"),
               "--preprocessing-config", str(config)])
    fitted = load_calibrated_predictor(artifact)
    assert fitted.metadata["input_contract"]["preprocessing"]["l_freq"] == .5
    task.main(["--task", "infer", "--session-dir", str(testing),
               "--model-artifact", str(artifact), "--output-dir", str(tmp_path / "infer")])
    result = read_json(tmp_path / "infer/summary.json")
    assert result["status"] in {"predicted", "abstained"}
    replay = fitted.load_inference_session(testing)
    expected = load_brainsync_inference_session(testing, preprocessing=spec)
    assert replay.preprocessing == spec
    np.testing.assert_array_equal(replay.X, expected.X)
    with pytest.raises(ValueError, match="input channels, reference"):
        fitted.predict_logit(load_brainsync_inference_session(testing))
    with pytest.raises(SystemExit):
        task.main(["--task", "infer", "--session-dir", str(testing),
                   "--model-artifact", str(artifact), "--output-dir", str(tmp_path / "override"),
                   "--preprocessing-config", str(config)])
    failure = read_json(tmp_path / "override/summary.json")
    assert "only allowed when calibrating" in failure["error"]["message"]
    changed = load_brainsync_sessions((roots[0],), preprocessing=spec)
    with pytest.raises(ValueError, match="executable P300 input contract"):
        fit_calibrated_predictor(changed)
    source = tmp_path / "original-band-source.pt"
    write_checkpoint(source, load_brainsync_sessions((roots[0],)), source_snapshot_sha256="f" * 64)
    assert load_calibration_source_checkpoint(changed, source)[3]["status"] == "incompatible"


def test_recovery_runner_saves_actual_frozen_ocular_operator(tmp_path, sessions):
    from data.ocular import eog_regression_from_record
    from experiments.run_brainsync_recovery import candidate_artifact
    from transfer.calibration_candidates import fit_calibration_candidate, waveform_lda_recipe

    roots, testing = sessions
    ocular = fit_brainsync_eog_sessions((roots[0],), reference_channel="Cz")
    dataset = load_brainsync_sessions((roots[0],), ocular_model=ocular)
    fitted = fit_calibration_candidate(dataset, waveform_lda_recipe(), np.ones(dataset.n_epochs, dtype=bool))
    predictor = candidate_artifact(dataset, fitted, None, {"status": "not_supplied"},
                                   device="cpu", plan_sha256="a" * 64)
    artifact = save_calibrated_predictor(tmp_path / "ocular-control.pt", predictor)
    restored = load_calibrated_predictor(artifact)
    request = restored.metadata["input_request"]
    frozen = eog_regression_from_record(request["ocular_model"])
    direct = load_brainsync_inference_session(testing, ocular_model=ocular)
    replay = load_brainsync_inference_session(testing, ocular_model=frozen)
    np.testing.assert_array_equal(direct.X, replay.X)
    np.testing.assert_array_equal(predictor.predict_logit(direct), restored.predict_logit(replay))


def test_source_numpy_unicode_ledger_loads_without_general_object_unpickling(tmp_path, sessions):
    from types import SimpleNamespace

    roots, _ = sessions
    dataset = load_brainsync_sessions(roots)
    source = tmp_path / "source.pt"
    write_checkpoint(source, dataset, source_snapshot_sha256="f" * 64)
    payload = torch.load(source, weights_only=True)
    payload["source_fit_ledger"] = {"training_participant_keys": [np.str_("public-person")]}
    torch.save(payload, source)
    original = source.read_bytes()
    _, loaded, _, gate = load_calibration_source_checkpoint(dataset, source)
    assert gate["status"] == "compatible"
    assert loaded["source_fit_ledger"]["training_participant_keys"] == ["public-person"]
    assert source.read_bytes() == original
    payload["unsupported_object"] = SimpleNamespace(value=1)
    torch.save(payload, source)
    assert load_calibration_source_checkpoint(dataset, source)[3]["status"] == "unreadable"


def test_personal_epoch_quality_is_frozen_and_reload_preserves_abstention(tmp_path, sessions):
    from data.artifact import FoldLocalArtifactPolicy

    roots, testing = sessions
    dataset = load_brainsync_sessions(roots)
    predictor = fit_calibrated_predictor(dataset, selector_mode="no_transfer",
        recipes=(CalibrationRecipe("qc_scratch", 1, 1e-3, initialization="scratch"),),
        epoch_quality_policy=FoldLocalArtifactPolicy(min_clean_epochs=2))
    assert predictor.metadata["epoch_quality"]["retained_training_rows"] <= dataset.n_epochs
    later = load_brainsync_inference_session(testing)
    later.X[:] *= 1000
    before = predictor.predict_logit(later)
    assert np.isnan(before).all()
    result = inference_record(predictor, later)
    assert result["status"] == "abstained" and result["evidence_trials"] == 0
    assert result["scheduled_trials"] == len(later.event_timeline.event_ids)
    assert result["model_unavailable_trials"] == later.n_epochs
    assert result["unavailable_trials"] == result["scheduled_trials"]
    restored = load_calibrated_predictor(save_calibrated_predictor(tmp_path / "qc.pt", predictor))
    np.testing.assert_array_equal(before, restored.predict_logit(later))
    output = tmp_path / "inspect_qc"
    model_task(["--task", "inspect", "--model-artifact", str(tmp_path / "qc.pt"), "--output-dir", str(output)])
    summary = read_json(output / "summary.json")
    exported = read_json(Path(summary["epoch_qc_model_path"]))
    assert exported["model"] == predictor.metadata["epoch_quality"]["model"]
    assert exported["digest"] == predictor.metadata["epoch_quality"]["model_digest"]


def test_quality_rejected_entire_r5_session_stays_in_operational_denominator(tmp_path):
    from transfer.calibration_candidates import score_calibration_candidate

    root = write_bids_session(tmp_path / "r5", session_id="r5", target=4,
        started_utc="2026-09-06T00:00:00Z", blocks=(tuple(range(1, 10)) * 5,))
    dataset = load_brainsync_sessions([root])
    score = score_calibration_candidate(dataset, np.full(dataset.n_epochs, np.nan),
        np.ones(dataset.n_epochs, dtype=bool), protocol="brainsync_operational_r5")
    assert score["operational_hit_r5"] == 0
    assert score["session_count"] == 1 and len(score["blocks"]) == 1
    assert not score["blocks"][0]["complete"]
    assert score["validation_auc"] is None and score["usable_validation_rows"] == 0


def test_all_raw_rejected_inference_writes_abstention_with_complete_request(tmp_path, sessions):
    import mne

    from data.brainsync_contract import validate_brainsync_bids_session
    from data.continuous_qc import ContinuousQCSpec, fit_continuous_qc
    from data.contract import BRAIN_SYNC_CAUSAL_DATA_CONTRACT
    from data.epochs import materialize_dataset_identity, preprocessing_spec_from_contract

    roots, testing = sessions
    first = validate_brainsync_bids_session(roots[0])
    initial = load_brainsync_sessions(roots)
    participant = materialize_dataset_identity(initial).record_for("P001").authority_key("source_or_global")
    raw = mne.io.read_raw_edf(first.bids.raw_path, preload=True, verbose=False)
    raw.set_channel_types(dict(zip(raw.ch_names, [kind.lower() for kind in first.bids.channel_types], strict=True)))
    model = fit_continuous_qc(raw, spec=ContinuousQCSpec(12, 1e-7, 10),
        fit_sample_mask=np.ones(raw.n_times, dtype=bool), fit_provenance={
            "participant_key": participant, "source_reference": first.bids.source_reference,
            "calibration_session_ids": [first.session_id], "calibration_end_timestamp_s": first.ended_timestamp_s})
    dataset = load_brainsync_sessions(roots, continuous_qc_model=model,
        preprocessing=preprocessing_spec_from_contract(BRAIN_SYNC_CAUSAL_DATA_CONTRACT))
    predictor = fit_calibrated_predictor(dataset)
    path = save_calibrated_predictor(tmp_path / "continuous.pt", predictor)
    later = validate_brainsync_bids_session(testing)
    signal = mne.io.read_raw_edf(later.bids.raw_path, preload=True, verbose=False)
    signal._data[3, 100:] += .02
    signal.export(later.bids.raw_path, overwrite=True, physical_range="auto", verbose=False)
    output = tmp_path / "abstention"
    model_task(["--task", "infer", "--session-dir", str(testing), "--model-artifact", str(path),
                "--output-dir", str(output), "--repetitions", "5"])
    record = read_json(output / "summary.json")
    assert record["status"] == "abstained" and record["prediction"] is None
    assert record["scheduled_trials"] == len(later.bids.stimuli)
    assert record["evidence_trials"] == 0 and len(record["by_repetition"]) == 5
    assert record["preprocessing_unavailable_trials"] == record["scheduled_trials"]
    assert record["model_unavailable_trials"] == 0
    assert record["participant_key"] == participant and not record["target_labels_used"]


def test_fixed_candidates_fit_only_inner_training_and_fresh_full_refit(sessions):
    from transfer.calibrated import select_calibration_candidate

    roots, _ = sessions
    dataset = load_brainsync_sessions(roots)
    recipe = TargetRecipe(
        "raw_lda",
        initialization="waveform_lda",
        weights="none",
        bn_buffers="source",
        bn_affine="frozen",
        dropout="off",
        epochs=1,
    )
    fitted, selection = select_calibration_candidate(
        dataset,
        (recipe,),
        calibration_mask=np.ones(dataset.n_epochs, dtype=bool),
    )
    assert selection["selection_evidence"]
    inner_training = dataset.metadata["session"].isin(("cal-0", "cal-1")).to_numpy()
    assert selection["recipes"][0]["fit_scope"]["training_rows"] == inner_training.sum()
    assert selection["refit_rows"] == dataset.n_epochs
    expected = target_shrinkage_scores(
        dataset.X, dataset.y, dataset.X, sfreq=dataset.preprocessing.sfreq
    )
    np.testing.assert_allclose(fitted.predict_logit(dataset.X), expected)
    changed = copy.deepcopy(dataset)
    changed.X[~inner_training] *= 100
    _, other = select_calibration_candidate(
        changed,
        (recipe,),
        calibration_mask=np.ones(dataset.n_epochs, dtype=bool),
    )
    assert (
        other["recipes"][0]["fitted_state_sha256"] == selection["recipes"][0]["fitted_state_sha256"]
    )


def test_new_selector_persists_scratch_actual_statistics(tmp_path, sessions):
    roots, testing = sessions
    dataset = load_brainsync_sessions(roots)
    predictor = fit_calibrated_predictor(
        dataset, selector_mode="no_transfer", selection_protocol="binary_auc",
        recipes=(CalibrationRecipe("scratch_probe", 1, 1e-3, initialization="scratch"),),
    )
    assert predictor.metadata["model_kind"] == "scratch_n2p3"
    np.testing.assert_allclose(
        predictor.state["input_mean"].numpy(), dataset.X.mean(axis=(0, 2), keepdims=True), atol=1e-11,
    )
    before = predictor.predict_logit(load_brainsync_inference_session(testing))
    restored = load_calibrated_predictor(save_calibrated_predictor(tmp_path / "scratch.pt", predictor))
    np.testing.assert_array_equal(restored.predict_logit(load_brainsync_inference_session(testing)), before)


def test_new_product_modes_consume_frozen_b0_rule_and_persist_it(tmp_path, sessions):
    from transfer.batch_cohort import declare_batch_cohort, freeze_batch_cohort_rule
    from transfer.binary import binary_time_split
    from transfer.calibrated import preflight_calibrated_predictor
    from transfer.target_experiments import build_target_tasks

    roots, _ = sessions
    dataset = load_brainsync_sessions(roots)
    fallback = preflight_calibrated_predictor(dataset, selector_mode="no_transfer")
    assert fallback["batch_rule"]["batch_size"] == 32
    assert fallback["batch_rule"]["selection_evidence"] is False
    assert {row["batch_size"] for row in fallback["candidates"]} == {32}
    with pytest.raises(ValueError, match="frozen B0 rule"):
        preflight_calibrated_predictor(dataset, selector_mode="no_transfer", batch_size=64)
    fixture = binary_dataset(400)
    split = binary_time_split(np.arange(400) * 2.0, np.arange(400) * 2.0 + .8,
                              calibration_trials=200, maximum_calibration_trials=200, tmin_s=-.2)
    manifest = build_target_tasks(fixture, {"p1": (np.arange(400), {200: split})}, stage="B0")
    person = fixture.identity_table.record_for("p1").authority_key()
    cohort = declare_batch_cohort([{"target_study": "synthetic_p300", **manifest}], {"synthetic_p300": [person]})
    rows = [dict(
        **cohort["cases"][0], budget_mode="epochs",
        batch_size=batch, arm=arm, score=0.9 if batch == 128 else 0.5, status="completed",
        elapsed_seconds=float(batch),
    ) for batch in (32, 128, 512) for arm in ("source", "scratch")]
    rule = freeze_batch_cohort_rule(rows, cohort)
    rule["evidence_level"] = "software_correctness"
    predictor = fit_calibrated_predictor(
        dataset, selector_mode="no_transfer", recipes=(CalibrationRecipe("scratch_probe", 1, 1e-3, initialization="scratch"),),
        batch_rule=rule,
    )
    assert predictor.metadata["batch_rule"] == rule
    assert predictor.metadata["recipe"]["batch_size"] == 128
    rule_path = tmp_path / "batch_rule.json"
    write_json(rule_path, rule)
    output = tmp_path / "preflight"
    model_task([
        "--task", "calibrate", "--selector-mode", "no_transfer", "--selection-protocol", "binary_auc",
        "--dry-run", "--batch-rule", str(rule_path), "--model-artifact", str(tmp_path / "unfitted.pt"),
        "--output-dir", str(output),
        *[argument for root in roots for argument in ("--session-dir", str(root))],
    ])
    assert read_json(output / "summary.json")["batch_rule"] == rule


def test_public_and_product_use_identical_recipe_and_full_refit_state(sessions):
    from transfer.calibrated import select_calibration_candidate

    roots, _ = sessions
    dataset = load_brainsync_sessions(roots)
    recipe = TargetRecipe("shared_scratch", initialization="scratch", epochs=1, seed=11, batch_size=32)
    public, _ = select_calibration_candidate(
        dataset, (recipe,), calibration_mask=np.ones(dataset.n_epochs, dtype=bool), selector_mode="no_transfer",
    )
    product = fit_calibrated_predictor(
        dataset, recipes=(recipe,), selector_mode="no_transfer", selection_protocol="binary_auc",
    )
    assert product.metadata["recipe"] == public.recipe.record()
    for name, tensor in public.state.items():
        torch.testing.assert_close(product.state[name], tensor, rtol=0, atol=0)


def test_operational_selector_rejects_non_r5_schedule(sessions):
    from transfer.calibrated import score_calibration_candidate

    roots, _ = sessions
    dataset = load_brainsync_sessions(roots)
    with pytest.raises(ValueError, match="45"):
        score_calibration_candidate(
            dataset, np.ones(dataset.n_epochs), np.ones(dataset.n_epochs, dtype=bool),
            protocol="brainsync_operational_r5",
        )


def test_raw_lda_uses_only_channels_observed_in_training_and_refits_them(sessions):
    from transfer.calibrated import fit_calibration_candidate

    roots, _ = sessions
    dataset = load_brainsync_sessions(roots)
    observed = np.ones(dataset.X.shape[:2], dtype=bool)
    observed[:54, 0] = False
    dataset = replace(dataset, trial_channel_mask=observed)
    recipe = TargetRecipe(
        "lda", initialization="waveform_lda", weights="none", bn_buffers="source",
        bn_affine="frozen", dropout="off", epochs=1,
    )
    model = fit_calibration_candidate(dataset, recipe, np.arange(dataset.n_epochs) < 54)
    np.testing.assert_array_equal(model.state["waveform_channels"].numpy(), np.arange(1, 7))
    expected = target_shrinkage_scores(dataset.X[:54, 1:], dataset.y[:54], dataset.X[54:, 1:],
                                       sfreq=dataset.preprocessing.sfreq)
    np.testing.assert_allclose(model.predict_logit(dataset.X[54:], observed[54:]), expected)
    observed[54:, 1] = False
    with pytest.raises(ValueError, match="selected channel"):
        model.predict_logit(dataset.X[54:], observed[54:])


def test_shared_candidate_rejects_eog_fitted_outside_its_training_scope(sessions):
    from transfer.calibrated import fit_calibration_candidate

    roots, _ = sessions
    ocular = fit_brainsync_eog_sessions(tuple(roots), reference_channel="Cz")
    dataset = load_brainsync_sessions(roots, ocular_model=ocular)
    recipe = TargetRecipe(
        "lda", initialization="waveform_lda", weights="none", bn_buffers="source",
        bn_affine="frozen", dropout="off", epochs=1,
    )
    with pytest.raises(ValueError, match="outside the declared calibration budget"):
        fit_calibration_candidate(dataset, recipe, np.arange(dataset.n_epochs) < 54)


def test_operational_r5_scores_sessions_equally_and_treats_ties_as_misses(tmp_path):
    from transfer.calibrated import score_calibration_candidate

    roots = [
        write_bids_session(
            tmp_path / f"session-{index}", session_id=f"session-{index}", target=index + 1,
            started_utc=f"2026-09-06T08:{index * 5:02d}:00Z",
            blocks=(tuple(range(1, 10)) * 5,) * (2 if index == 0 else 1),
        ) for index in range(2)
    ]
    dataset = load_brainsync_sessions(roots)
    logits = dataset.y.astype(float)
    logits[dataset.metadata["session"] == "session-1"] = 0
    record = score_calibration_candidate(
        dataset, logits, np.ones(dataset.n_epochs, dtype=bool), protocol="brainsync_operational_r5",
    )
    assert record["primary_score"] == 0.5
    assert [block["hit"] for block in record["blocks"]] == [1, 1, 0]
    assert record["blocks"][-1]["prediction"] is None


def test_incompatible_source_new_selector_records_gate_and_legal_fallback(tmp_path, sessions):
    roots, testing = sessions
    dataset = load_brainsync_sessions(roots)
    projected = adapt_observed_montage(
        dataset, tuple(name for name in dataset.channel_names if name != "CZ"), reference_channel="Cz",
    )
    source = write_checkpoint(tmp_path / "source.pt", projected, source_snapshot_sha256="f" * 64)
    predictor = fit_calibrated_predictor(
        dataset, source_checkpoint=source, selector_mode="transfer_only", selection_protocol="binary_auc",
        recipes=(CalibrationRecipe("source_probe", 1, 1e-3),),
    )
    selection = predictor.metadata["selection"]
    assert selection["source_checkpoint_gate"]["status"] == "incompatible"
    assert selection["source_checkpoint_gate"]["message"]
    assert selection["recipes"][0]["status"] == "failed"
    assert not selection["selection_evidence"]
    assert not predictor.metadata["source_pretraining_used"]
    artifact = save_calibrated_predictor(tmp_path / "fallback.pt", predictor)
    target = load_brainsync_inference_session(testing)
    np.testing.assert_array_equal(load_calibrated_predictor(artifact).predict_logit(target), predictor.predict_logit(target))


def test_new_selector_dry_run_from_outside_repository_never_fits_or_writes_model(tmp_path, sessions):
    roots, _ = sessions
    output = tmp_path / "plan"
    artifact = tmp_path / "unfitted.pt"
    process = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "experiments" / "brainsync_model_task.py"),
         "--task", "calibrate", "--selector-mode", "five_candidates", "--dry-run",
         "--model-artifact", str(artifact), "--output-dir", str(output),
         *[argument for root in roots for argument in ("--session-dir", str(root))]],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    plan = read_json(output / "summary.json")
    assert plan["candidate_count"] == 5
    assert plan["data_fitted"] is False
    assert plan["inner_split"]["training_rows"] == 54
    assert plan["inner_split"]["validation_rows"] == 27
    assert not artifact.exists()


@pytest.mark.parametrize(
    "build_source,expected_status,required_details",
    [
        pytest.param(lambda path, dataset: None, "unreadable", ("message",), id="missing"),
        pytest.param(
            lambda path, dataset: torch.save({"schema": "unsupported"}, path),
            "incompatible",
            ("message",),
            id="schema",
        ),
        pytest.param(
            partial(write_checkpoint, source_snapshot_sha256="f" * 64),
            "incompatible",
            ("message",),
            id="native-source-for-cz-input",
        ),
        pytest.param(_write_cz_checkpoint, "compatible", ("checkpoint_sha256",), id="cz-source"),
    ],
)
def test_dry_run_records_source_gate_and_plans_eog_without_any_fit(
    tmp_path,
    sessions,
    monkeypatch,
    build_source,
    expected_status,
    required_details,
):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    from transfer.subject_adapter import SubjectAdapter

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run attempted a fitted transform or model")

    roots, _ = sessions
    source = tmp_path / "source.pt"
    build_source(source, load_brainsync_sessions(roots))
    monkeypatch.setattr(SubjectAdapter, "fit", forbidden)
    monkeypatch.setattr(LinearDiscriminantAnalysis, "fit", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr("experiments.brainsync_model_task.fit_brainsync_eog_sessions", forbidden)
    output = tmp_path / "dry-run"
    artifact = tmp_path / "unfitted.pt"
    model_task(
        [
            "--task",
            "calibrate",
            "--selector-mode",
            "five_candidates",
            "--selection-protocol",
            "binary_auc",
            "--dry-run",
            "--source-checkpoint",
            str(source),
            "--ocular-correction",
            "--reference-channel",
            "Cz",
            "--model-artifact",
            str(artifact),
            "--output-dir",
            str(output),
            *[argument for root in roots for argument in ("--session-dir", str(root))],
        ]
    )
    plan = read_json(output / "summary.json")
    gate = plan["source_checkpoint_gate"]
    assert gate["status"] == expected_status
    assert all(gate[field] for field in required_details)
    assert plan["inner_split"]["legal"]
    assert plan["planned_ocular"]["fit_sessions"] == ["cal-0", "cal-1"]
    assert plan["planned_ocular"]["fitted"] is False
    assert plan["full_calibration_lda_gate"]["legal"]
    assert not artifact.exists()


def test_product_preflight_without_legal_inner_sessions_plans_full_lda_fallback(sessions):
    from transfer.calibrated import preflight_calibrated_predictor

    roots, _ = sessions
    dataset = load_brainsync_sessions(roots[:1])
    plan = preflight_calibrated_predictor(dataset, selector_mode="five_candidates", planned_ocular=True)
    assert not plan["inner_split"]["legal"]
    assert not plan["inner_selection_planned"]
    assert plan["fallback_planned"] and plan["execution_ready"]
    assert plan["planned_ocular"]["fit_sessions"] == ["cal-0"]
    assert plan["planned_ocular"]["fit_rows"] == dataset.n_epochs


@pytest.fixture
def sessions(tmp_path):
    roots = []
    for index, target in enumerate((1, 3, 5)):
        roots.append(write_bids_session(
            tmp_path / f"cal-{index}", session_id=f"cal-{index}", target=target,
            started_utc=f"2026-09-06T08:{index * 3:02d}:00Z",
            blocks=(tuple(range(1, 10)) * 3,),
        ))
    testing = write_bids_session(
        tmp_path / "test", session_id="new-test", target=7,
        started_utc="2026-09-06T08:12:00Z", blocks=(tuple(range(1, 10)) * 3,),
    )
    return roots, testing


@pytest.mark.parametrize(
    ("fit_session_count", "with_source", "blocked"),
    [(3, True, True), (2, True, False), (3, False, False)],
    ids=["validation-qc-rejected", "inner-qc-source-accepted", "full-qc-fixed-lda-accepted"],
)
def test_legacy_real_cli_checks_raw_quality_scope_before_training(
    tmp_path, sessions, monkeypatch, fit_session_count, with_source, blocked
):
    import mne

    import transfer.calibrated as calibrated
    from data.brainsync_contract import validate_brainsync_bids_session
    from data.continuous_qc import ContinuousQCSpec, fit_continuous_qc, save_continuous_qc
    from data.contract import BRAIN_SYNC_CAUSAL_DATA_CONTRACT
    from data.epochs import materialize_dataset_identity, preprocessing_spec_from_contract

    roots, testing = sessions
    preprocessing = preprocessing_spec_from_contract(BRAIN_SYNC_CAUSAL_DATA_CONTRACT)
    initial = load_brainsync_sessions(roots, preprocessing=preprocessing)
    fitting = [validate_brainsync_bids_session(root) for root in roots[:fit_session_count]]
    raws = []
    for session in fitting:
        raw = mne.io.read_raw_edf(session.bids.raw_path, preload=True, verbose=False)
        raw.set_channel_types(dict(zip(
            raw.ch_names, [kind.lower() for kind in session.bids.channel_types], strict=True)))
        raws.append(raw)
    raw = mne.concatenate_raws(raws, verbose=False)
    quality = fit_continuous_qc(
        raw, spec=ContinuousQCSpec(12, 1e-7, 10, channel_local=False),
        fit_sample_mask=np.ones(raw.n_times, dtype=bool), fit_provenance={
            "participant_key": materialize_dataset_identity(initial).authority_keys()[0],
            "source_reference": fitting[0].bids.source_reference,
            "calibration_session_ids": [session.session_id for session in fitting],
            "calibration_end_timestamp_s": max(session.ended_timestamp_s for session in fitting),
        },
    )
    dataset = load_brainsync_sessions(roots, preprocessing=preprocessing, continuous_qc_model=quality)
    source = write_checkpoint(tmp_path / "scope-source.pt", dataset, source_snapshot_sha256="f" * 64)
    source = source if with_source else None
    plan = preflight_calibrated_predictor(dataset, source_checkpoint=source)
    assert plan["execution_ready"] is not blocked
    attempts = []

    def stop_before_training(*args, **kwargs):
        attempts.append(kwargs["config"])
        raise RuntimeError("accepted_scope_stop_before_training")

    monkeypatch.setattr(calibrated, "SubjectAdapter", stop_before_training)
    profile = save_continuous_qc(tmp_path / "global-quality.json", quality)
    from dataclasses import asdict

    config = tmp_path / "causal.json"
    write_json(config, asdict(preprocessing))
    output, artifact = tmp_path / "scope-result", tmp_path / "scope-model.pt"
    arguments = ["--task", "calibrate", "--model-artifact", str(artifact),
                 "--output-dir", str(output), "--preprocessing-config", str(config),
                 "--continuous-qc-model", str(profile)]
    arguments.extend(value for root in roots for value in ("--session-dir", str(root)))
    if source is not None:
        arguments.extend(("--source-checkpoint", str(source)))
        with pytest.raises(SystemExit):
            model_task(arguments)
        error = read_json(output / "summary.json")["error"]
        if blocked:
            assert not attempts
            assert "continuous QC used sessions outside" in error["message"]
        else:
            assert len(attempts) == 1
            assert error["message"] == "accepted_scope_stop_before_training"
        assert not artifact.exists()
    else:
        model_task(arguments)
        saved = load_calibrated_predictor(artifact)
        assert saved.metadata["input_request"]["continuous_qc_model"] == quality.record()
        assert saved.metadata["selection"]["mode"] == "fixed_fallback_no_recipe_selection"
        saved.load_inference_session(testing)
        assert not attempts


@pytest.mark.parametrize("full_recipe", [False, True], ids=["budget-recipe", "explicit-weight-decay"])
def test_legacy_recipe_configuration_is_independent_of_epoch_quality_route(
    tmp_path, sessions, monkeypatch, full_recipe
):
    from dataclasses import asdict

    import transfer.calibrated as calibrated
    import transfer.calibration_candidates as candidates

    roots, _ = sessions
    dataset = load_brainsync_sessions(roots)
    source = write_checkpoint(tmp_path / "route-source.pt", dataset, source_snapshot_sha256="f" * 64)
    recipe = (TargetRecipe("probe", input_statistics="source", bn_buffers="source",
                           batch_size=64, seed=20260906, epochs=1, weight_decay=.031)
              if full_recipe else CalibrationRecipe("probe", 1, 1e-4))
    configurations = []

    def stop_before_training(*args, **kwargs):
        config = asdict(kwargs["config"])
        for key in ("input_mean", "input_std"):
            config[key] = np.asarray(config[key]).tolist()
        configurations.append(config)
        raise RuntimeError("configuration_captured")

    monkeypatch.setattr(calibrated, "SubjectAdapter", stop_before_training)
    monkeypatch.setattr(candidates, "SubjectAdapter", stop_before_training)
    for quality in (None, FoldLocalArtifactPolicy(min_clean_epochs=2)):
        with pytest.raises(RuntimeError, match="configuration_captured"):
            fit_calibrated_predictor(dataset, source_checkpoint=source, recipes=(recipe,),
                                     epoch_quality_policy=quality)
    assert configurations[0] == configurations[1]
    assert configurations[0]["input_statistics"] == "source"
    assert configurations[0]["batchnorm_buffers"] == "source"
    assert configurations[0]["weight_decay"] == (.031 if full_recipe else 1e-4)


@pytest.mark.parametrize("quality", [False, True], ids=["plain-route", "quality-route"])
@pytest.mark.parametrize("conflict", ["initialization", "bn_affine", "dropout", "bn_buffers", "refit", "updates", "transfer_policy"])
def test_legacy_conflicting_recipe_is_rejected_before_either_fit_route(
    tmp_path, sessions, monkeypatch, quality, conflict
):
    import transfer.calibrated as calibrated
    import transfer.calibration_candidates as candidates

    roots, _ = sessions
    dataset = load_brainsync_sessions(roots)
    source = write_checkpoint(tmp_path / "conflict-source.pt", dataset, source_snapshot_sha256="f" * 64)
    recipe = CalibrationRecipe("probe", 1, 1e-4)
    extra = {}
    if conflict in {"initialization", "bn_affine", "dropout", "bn_buffers"}:
        value = {"initialization": "scratch", "bn_affine": "frozen", "dropout": "off", "bn_buffers": "calibration"}[conflict]
        recipe = replace(recipe, **{conflict: value})
    elif conflict in {"refit", "updates"}:
        recipe = TargetRecipe("probe", input_statistics="source", bn_buffers="source",
                              batch_size=64, seed=20260906, epochs=1,
                              **({"refit_full_prefix": True} if conflict == "refit" else {"max_updates": 1}))
    else:
        extra["transfer_policy"] = TargetRecipe("policy")

    def forbidden(*args, **kwargs):
        raise AssertionError("conflicting legacy policy reached fitting")

    monkeypatch.setattr(calibrated, "SubjectAdapter", forbidden)
    monkeypatch.setattr(candidates, "SubjectAdapter", forbidden)
    with pytest.raises(ValueError, match="legacy"):
        fit_calibrated_predictor(
            dataset, source_checkpoint=source, recipes=(recipe,),
            epoch_quality_policy=FoldLocalArtifactPolicy(min_clean_epochs=2) if quality else None,
            **extra,
        )


def test_legacy_noop_frozen_quality_preserves_fit_state_recipe_and_reload(tmp_path, sessions):
    roots, testing = sessions
    dataset = load_brainsync_sessions(roots)
    source = write_checkpoint(tmp_path / "noop-source.pt", dataset, source_snapshot_sha256="f" * 64)
    training = dataset.metadata["session"].isin(("cal-0", "cal-1")).to_numpy()
    quality = fit_epoch_quality(dataset, training, FoldLocalArtifactPolicy(min_clean_epochs=2))
    # Explicit synthetic permissive thresholds isolate execution routing, not QC efficacy.
    quality = replace(quality, ptp_thresholds=np.full(dataset.n_channels, 1e6),
                      flat_std_thresholds=np.zeros(dataset.n_channels),
                      global_scale_log_robust_std=1e6)
    transformed = quality.transform(dataset.X)
    assert not transformed.drop_epoch_mask.any()
    assert transformed.trial_channel_mask.all()
    np.testing.assert_array_equal(transformed.X, dataset.X)
    recipe = CalibrationRecipe("noop", 1, 1e-4)
    plain = fit_calibrated_predictor(dataset, source_checkpoint=source, recipes=(recipe,))
    frozen = fit_calibrated_predictor(dataset, source_checkpoint=source, recipes=(recipe,),
                                      epoch_quality_model=quality)
    assert plain.metadata["recipe"] == frozen.metadata["recipe"]
    for name in plain.state:
        torch.testing.assert_close(plain.state[name], frozen.state[name], rtol=0, atol=0)
    assert frozen.metadata["epoch_quality"]["lifecycle"] == "frozen_explicit"
    assert frozen.epoch_quality.record() == quality.record()
    restored = load_calibrated_predictor(save_calibrated_predictor(tmp_path / "noop.pt", frozen))
    assert restored.metadata["recipe"] == frozen.metadata["recipe"]
    assert restored.epoch_quality.record() == quality.record()
    target = restored.load_inference_session(testing)
    before = copy.deepcopy(restored.state)
    prediction = inference_record(restored, target)
    assert prediction == inference_record(frozen, target)
    np.testing.assert_array_equal(plain.predict_logit(target), restored.predict_logit(target))
    for name in before:
        torch.testing.assert_close(before[name], restored.state[name], rtol=0, atol=0)
    assert restored.epoch_quality.record() == quality.record()


def test_explicit_legacy_configuration_preserves_historical_full_refit(tmp_path, sessions):
    from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig

    roots, _ = sessions
    dataset = load_brainsync_sessions(roots)
    source = write_checkpoint(tmp_path / "historical-source.pt", dataset, source_snapshot_sha256="f" * 64)
    current = fit_calibrated_predictor(dataset, source_checkpoint=source,
                                     recipes=(CalibrationRecipe("one-epoch", 1, 1e-4),))
    trunk, _, statistics, gate = load_calibration_source_checkpoint(dataset, source)
    assert gate["status"] == "compatible"
    historical = SubjectAdapter(trunk, config=SubjectAdapterConfig(
        head_kind="full_fine", epochs=1, lr=1e-4, batch_size=64, seed=20260906,
        val_group_fraction=None, refit_full_prefix=False, freeze_batchnorm_stats=True,
        input_statistics="source", input_mean=statistics[0], input_std=statistics[1],
    ))
    observed = np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2])
    historical.fit(dataset.X, dataset.y, prefix_mask=np.ones(dataset.n_epochs, dtype=bool),
                   trial_channel_mask=observed)
    for name, tensor in historical.trunk.state_dict().items():
        torch.testing.assert_close(tensor, current.state[f"trunk.{name}"], rtol=0, atol=0)
    mean, std = historical.fitted_input_statistics()
    np.testing.assert_array_equal(mean, current.state["input_mean"].numpy())
    np.testing.assert_array_equal(std, current.state["input_std"].numpy())
    assert current.metadata["recipe"] == current.metadata["selection"]["refit_recipe"]


def test_fixed_inner_training_is_invariant_to_validation_with_quality_policy(sessions):
    from data.qc_features import compute_epoch_qc_features
    from transfer.calibration_candidates import fit_calibration_candidate

    roots, _ = sessions
    dataset = load_brainsync_sessions(roots)
    training = dataset.metadata["session"].isin(("cal-0", "cal-1")).to_numpy()
    values, labels = dataset.X.copy(), dataset.y.copy()
    values[~training] *= 1.2
    labels[~training] = dataset.metadata.loc[~training, "candidate_id"].astype(int).eq(7).to_numpy()
    metadata = dataset.metadata.copy()
    metadata.loc[~training, "target_candidate_id"] = "7"
    target_ids = dataset.event_timeline.target_candidate_ids.copy()
    target_ids[np.asarray(dataset.event_timeline.session_ids) == "cal-2"] = "7"
    changed = replace(dataset, X=values, y=labels,
                      metadata=metadata,
                      event_timeline=replace(dataset.event_timeline, target_candidate_ids=target_ids),
                      qc_features=compute_epoch_qc_features(values, channel_mask=dataset.channel_mask))
    recipe = TargetRecipe("fixed-inner", initialization="scratch", epochs=2, batch_size=32, seed=19)
    options = {"validation_mask": ~training, "validation_behavior": "diagnostics_only",
               "epoch_quality_policy": FoldLocalArtifactPolicy(min_clean_epochs=2)}
    first = fit_calibration_candidate(dataset, recipe, training, **options)
    second = fit_calibration_candidate(changed, recipe, training, **options)
    assert first.epoch_quality.record() == second.epoch_quality.record()
    assert first.fit_history["training_data_sha256"] == second.fit_history["training_data_sha256"]
    for name in first.state:
        torch.testing.assert_close(first.state[name], second.state[name], rtol=0, atol=0)


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
    source = write_checkpoint(tmp_path / "source.pt", dataset, source_snapshot_sha256="f" * 64)
    predictor = fit_calibrated_predictor(
        dataset, source_checkpoint=source,
        recipes=(CalibrationRecipe("small", 1, 1e-4), CalibrationRecipe("larger", 2, 1e-3)),
        batch_size=64,
    )
    selection = predictor.metadata["selection"]
    assert selection["training_sessions"] == ["cal-0", "cal-1"]
    assert selection["validation_sessions"] == ["cal-2"]
    assert selection["refit_rows"] == dataset.n_epochs
    assert selection["fit_scope"]["training_rows"] == dataset.n_epochs
    assert selection["fit_scope"]["validation_rows"] == 0
    assert selection["fit_scope"]["prefix_mask_explicit"]
    assert len(selection["refit_history"]["train_losses"]) == selection["selected"]["selected_epochs"]
    for recipe in selection["recipes"]:
        assert recipe["fit_history"]["train_losses"]
        assert len(recipe["fit_history"]["train_losses"]) == len(recipe["fit_history"]["val_losses"])
    assert predictor.metadata["source_pretraining_used"]
    np.testing.assert_array_equal(predictor.state["input_std"].numpy(), np.ones((1, 7, 1)))
    target = load_brainsync_inference_session(testing)
    before = predictor.predict_logit(target)
    saved = save_calibrated_predictor(tmp_path / "adapted.pt", predictor)
    restored = load_calibrated_predictor(saved)
    np.testing.assert_array_equal(restored.predict_logit(target), before)
    from experiments.brainsync_model_task import _model_summary

    assert _model_summary(restored)["internal_validation"] == selection
    assert torch.load(saved, weights_only=True)["metadata"]["model_kind"] == "source_initialized_n2p3"


@pytest.mark.parametrize("weights", ["spatial", "temporal", "adapter"])
def test_structural_personal_recipes_refit_and_reload_without_losing_their_model(tmp_path, sessions, weights):
    from models.adapters import FeatureDomainTransfer

    roots, testing = sessions
    dataset = load_brainsync_sessions(roots)
    source = write_checkpoint(tmp_path / "source.pt", dataset, source_snapshot_sha256="f" * 64)
    recipe = TargetRecipe("structural", weights=weights, batch_size=32, max_updates=2,
                          bn_affine="frozen", dropout="off")
    predictor = fit_calibrated_predictor(dataset, source_checkpoint=source,
        recipes=(recipe,), selector_mode="transfer_only", batch_size=32)
    assert predictor.metadata["recipe"] == recipe.record()
    assert isinstance(predictor.trunk, FeatureDomainTransfer) == (weights == "adapter")
    target = load_brainsync_inference_session(testing)
    before = predictor.predict_logit(target)
    saved = save_calibrated_predictor(tmp_path / "personal.pt", predictor)
    restored = load_calibrated_predictor(saved)
    np.testing.assert_array_equal(restored.predict_logit(target), before)
    assert restored.metadata["selection"]["refit_history"]["parameter_groups"]
    if weights == "adapter":
        altered = torch.load(saved, weights_only=True)
        altered["metadata"]["recipe"]["target_adapter_config"]["target_kernel_size"] += 2
        from transfer.calibrated import CalibratedPredictor

        with pytest.raises(ValueError, match="architecture"):
            CalibratedPredictor(altered["metadata"], altered["state"])


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
    wrong_person_dir = write_bids_session(
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


@pytest.mark.parametrize(
    "n_sessions,reference,ocular,n_channels",
    [(3, "Cz", False, 6), (3, "Cz", True, 6), (1, None, False, 7)],
    ids=["cz", "cz-eog", "one-session-native"],
)
def test_fixed_lda_cli_calibration_inference_and_inspection(
    tmp_path, sessions, n_sessions, reference, ocular, n_channels,
):
    roots, testing = sessions
    artifact = tmp_path / "personal.pt"
    calibration_output = tmp_path / "calibration"
    arguments = ["--task", "calibrate", "--model-artifact", str(artifact),
                 "--output-dir", str(calibration_output),
                 *[arg for root in roots[:n_sessions] for arg in ("--session-dir", str(root))]]
    if reference:
        arguments.extend(["--reference-channel", reference])
    if ocular:
        arguments.append("--ocular-correction")
    model_task(arguments)
    result = read_json(calibration_output / "summary.json")
    assert result["status"] == "completed"
    assert result["model_kind"] == "target_only_shrinkage_lda"
    assert result["ocular_correction_applied"] is ocular
    assert len(result["calibration_sessions"]) == n_sessions
    assert result["target_trials"] == 3 * n_sessions
    assert result["selected_recipe"] is None
    assert result["internal_validation"] == {
        "mode": "fixed_fallback_no_recipe_selection", "validation_auc": None,
    }
    assert load_calibrated_predictor(artifact).metadata["input_contract"]["n_channels"] == n_channels
    pending = read_json(testing / "session.json")
    pending["target_label"] = {"status": "pending"}
    pending["experiment"]["thought_digit"] = None
    pending["experiment"]["target_label_status"] = "pending_post_experiment_confirmation"
    write_json(testing / "session.json", pending)
    inference_output = tmp_path / "inference"
    model_task(["--task", "infer", "--model-artifact", str(artifact),
                "--output-dir", str(inference_output), "--session-dir", str(testing)])
    inference = read_json(inference_output / "summary.json")
    assert inference["status"] in {"predicted", "abstained"}
    assert inference["requested_repetitions"] == len(inference["by_repetition"]) == 3
    assert inference["target_labels_used"] is False
    assert set(inference["candidate_scores"]) == set(map(str, range(1, 10)))
    inspected = tmp_path / "inspected"
    model_task(["--task", "inspect", "--model-artifact", str(artifact), "--output-dir", str(inspected)])
    inspected_record = read_json(inspected / "summary.json")
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


@pytest.fixture
def incompatible_source(tmp_path, sessions):
    roots, _ = sessions
    dataset = load_brainsync_sessions(roots)
    source = _write_cz_checkpoint(tmp_path / "source.pt", dataset)
    return roots, dataset, source


def _assert_incompatible_source(dataset, source, **quality_options):
    plan = preflight_calibrated_predictor(dataset, source_checkpoint=source, **quality_options)
    assert not plan["execution_ready"]
    assert plan["source_checkpoint_gate"]["status"] == "incompatible"
    with pytest.raises(ValueError, match="source checkpoint is incompatible"):
        fit_calibrated_predictor(dataset, source_checkpoint=source, **quality_options)


@pytest.mark.parametrize(
    "quality_options",
    [{}, {"epoch_quality_policy": FoldLocalArtifactPolicy(min_clean_epochs=2)}],
    ids=["no-quality", "unfitted-epoch-policy"],
)
def test_source_gate_precedes_epoch_policy_fitting(incompatible_source, quality_options):
    _, dataset, source = incompatible_source
    _assert_incompatible_source(dataset, source, **quality_options)


def test_source_gate_with_applied_continuous_quality(incompatible_source):
    roots, _, source = incompatible_source
    dataset = load_brainsync_sessions(roots, continuous_qc_model=fit_raw_quality(roots[0]))
    _assert_incompatible_source(dataset, source)


def test_source_gate_with_frozen_epoch_quality(incompatible_source):
    _, dataset, source = incompatible_source
    quality = fit_epoch_quality(
        dataset,
        dataset.metadata["session"].eq("cal-0").to_numpy(),
        FoldLocalArtifactPolicy(min_clean_epochs=2),
    )
    _assert_incompatible_source(dataset, source, epoch_quality_model=quality)


def test_legacy_failed_source_fits_never_fit_a_fallback_lda(sessions, monkeypatch):
    import transfer.calibration_candidates as candidates

    roots, _ = sessions
    dataset = load_brainsync_sessions(roots)
    attempts = []

    def fail_source(dataset, recipe, *args, **kwargs):
        attempts.append(recipe.initialization)
        raise ValueError("source optimizer probe failure")

    monkeypatch.setattr(candidates, "fit_calibration_candidate", fail_source)
    with pytest.raises(ValueError, match="legacy source calibration failed.*optimizer probe failure"):
        candidates.select_calibration_candidate(
            dataset, candidates.calibration_candidate_recipes("legacy_product"),
            calibration_mask=np.ones(dataset.n_epochs, dtype=bool), selector_mode="legacy_product",
        )
    assert attempts == ["source", "source"]


@pytest.fixture
def continuous_eog_profile(tmp_path, sessions):
    from data.continuous_qc import save_continuous_qc

    roots, _ = sessions
    model = fit_raw_quality(roots[0], include_eog=True)
    return model, save_continuous_qc(tmp_path / "continuous.json", model)


def _check_fixed_lda_cli_lifecycle(tmp_path, sessions, options):
    roots, testing = sessions
    arguments = [
        "--task",
        "calibrate",
        *options,
        *[value for root in roots for value in ("--session-dir", str(root))],
    ]
    model_task(
        [
            *arguments,
            "--dry-run",
            "--model-artifact",
            str(tmp_path / "dry.pt"),
            "--output-dir",
            str(tmp_path / "dry"),
        ]
    )
    plan = read_json(tmp_path / "dry/summary.json")
    assert plan["execution_ready"] and not plan["inner_selection_planned"]
    assert plan["planned_ocular"]["fit_sessions"] == [
        read_json(root / "session.json")["session_id"] for root in roots
    ]
    artifact = tmp_path / "personal.pt"
    model_task(
        [*arguments, "--model-artifact", str(artifact), "--output-dir", str(tmp_path / "fit")]
    )
    predictor = load_calibrated_predictor(artifact)
    selection = predictor.metadata["selection"]
    assert predictor.metadata["model_kind"] == "target_only_shrinkage_lda"
    assert not selection["selection_evidence"] and selection["validation_auc"] is None
    assert selection["recipes"] == [] and "fallback" not in selection
    assert selection["fit_execution"]["fit_calls"] == 1
    before = artifact.read_bytes()
    model_task(
        [
            "--task",
            "infer",
            "--model-artifact",
            str(artifact),
            "--session-dir",
            str(testing),
            "--output-dir",
            str(tmp_path / "infer"),
        ]
    )
    result = read_json(tmp_path / "infer/summary.json")
    assert result["status"] in {"predicted", "abstained"} and not result["target_labels_used"]
    assert artifact.read_bytes() == before
    return predictor


@pytest.mark.parametrize(
    "epoch_options", [[], ["--epoch-qc"]], ids=["continuous-only", "fit-epoch-qc"]
)
def test_fixed_lda_fits_full_scope_eog_with_declared_epoch_policy(
    tmp_path, sessions, continuous_eog_profile, epoch_options
):
    _, profile = continuous_eog_profile
    _check_fixed_lda_cli_lifecycle(
        tmp_path,
        sessions,
        [
            "--continuous-qc-model",
            str(profile),
            "--reference-mode",
            "native",
            "--ocular-correction",
            *epoch_options,
        ],
    )


def test_fixed_lda_reuses_frozen_eog_and_epoch_quality_models(
    tmp_path, sessions, continuous_eog_profile
):
    from data.ocular import save_eog_regression
    from data.personal_quality import PersonalEpochQualityPolicy
    from transfer.quality import save_epoch_quality

    roots, _ = sessions
    continuous, profile = continuous_eog_profile
    ocular = fit_brainsync_eog_sessions(tuple(roots), continuous_qc_model=continuous)
    dataset = load_brainsync_sessions(roots, continuous_qc_model=continuous, ocular_model=ocular)
    frozen = fit_epoch_quality(
        dataset, np.ones(dataset.n_epochs, dtype=bool), PersonalEpochQualityPolicy()
    )
    predictor = _check_fixed_lda_cli_lifecycle(
        tmp_path,
        sessions,
        [
            "--continuous-qc-model",
            str(profile),
            "--reference-mode",
            "native",
            "--ocular-model",
            str(save_eog_regression(tmp_path / "eog.json", ocular)),
            "--epoch-qc-model",
            str(save_epoch_quality(tmp_path / "epoch.json", frozen)),
        ],
    )
    assert predictor.epoch_quality.record() == frozen.record()


def test_neural_eog_is_frozen_from_inner_training_sessions_only(tmp_path, sessions):
    roots, testing = sessions
    ocular = fit_brainsync_eog_sessions(tuple(roots[:2]), reference_channel="Cz")
    calibration = load_brainsync_sessions(roots, ocular_model=ocular)
    source = write_checkpoint(tmp_path / "source.pt", calibration, source_snapshot_sha256="e" * 64)
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
    result = read_json(output / "summary.json")
    assert result["status"] == "failed"
    assert result["error"]["type"] == "ValueError"
    assert "at least two" in result["error"]["message"]
    assert not artifact.exists()
